Master Cleanup Script for the ASF Medical Research Synthesizer Codebase.

This script runs all the cleanup scripts to perform a comprehensive cleanup of the codebase.

Usage:
    python -m asf.medical.scripts.master_cleanup [--dry-run]

import os
import sys
import logging
import argparse
import importlib
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("MasterCleanup")

CLEANUP_SCRIPTS = [
    "asf.medical.scripts.deep_cleanup",
    "asf.medical.scripts.standardize_caching",
    "asf.medical.scripts.standardize_db_access",
    "asf.medical.scripts.cleanup_codebase",  # Original cleanup script
    "asf.medical.scripts.standardize_service_naming",
    "asf.medical.scripts.standardize_imports",
    "asf.medical.scripts.fix_unused_imports",
    "asf.medical.scripts.fix_docstrings",
]

def run_cleanup_script(script_name: str, dry_run: bool = False, directory: Optional[str] = None) -> bool:
    """Run a cleanup script.

    Args:
        script_name: Name of the script module to run
        dry_run: Whether to run in dry-run mode
        directory: Directory to process

    Returns:
        True if the script ran successfully, False otherwise
    """
    try:
        logger.info(f"Running {script_name}...")

        script_module = importlib.import_module(script_name)

        main_func = getattr(script_module, "main", None)
        if main_func is None:
            logger.error(f"No main function found in {script_name}")
            return False

        # Save original argv
        original_argv = sys.argv.copy()

        # Set up new argv
        if script_name.endswith("standardize_service_naming") or \
           script_name.endswith("standardize_imports") or \
           script_name.endswith("fix_unused_imports") or \
           script_name.endswith("fix_docstrings") or \
           script_name.endswith("standardize_error_handling"):
            # These scripts use positional arguments
            sys.argv = [script_name, directory]
            if dry_run:
                pass  # These scripts don't support dry-run
            else:
                sys.argv.append("--fix")
        else:
            # Original scripts use --directory
            sys.argv = [script_name]
            if dry_run:
                sys.argv.append("--dry-run")
            if directory:
                sys.argv.extend(["--directory", directory])

        # Run the script
        main_func()

        # Restore original argv
        sys.argv = original_argv

        logger.info(f"Finished running {script_name}")
        return True
    except Exception as e:
        logger.error(f"Error running {script_name}: {str(e)}")
        return False

def main():
    Main function to run the script.
    
    Parses command-line arguments and runs the specified cleanup scripts.
    parser = argparse.ArgumentParser(description="Master cleanup script for the ASF Medical Research Synthesizer codebase")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually modify files, just show what would be done")
    parser.add_argument("--directory", default=None, help="Directory to process (default: asf/medical root directory)")
    parser.add_argument("--scripts", nargs="+", help="Specific scripts to run (default: all)")
    args = parser.parse_args()

    if args.directory:
        directory = args.directory
    else:
        # Default to the asf/medical directory
        directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Ensure directory exists
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        sys.exit(1)

    logger.info(f"Running cleanup scripts on directory: {directory}")

    scripts_to_run = args.scripts if args.scripts else CLEANUP_SCRIPTS

    successful_scripts = 0
    for script_name in scripts_to_run:
        if run_cleanup_script(script_name, args.dry_run, directory):
            successful_scripts += 1

    logger.info(f"Successfully ran {successful_scripts}/{len(scripts_to_run)} cleanup scripts")

    if args.dry_run:
        logger.info("This was a dry run. No files were actually modified.")

if __name__ == "__main__":
    main()
