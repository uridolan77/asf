"""Run Dramatiq Workers for the Medical Research Synthesizer.

This script starts Dramatiq workers for processing background tasks such as
export operations and ML inference tasks. It configures the number of worker
processes and threads based on command-line arguments or defaults.

The workers process tasks from the following modules:
- asf.medical.tasks.export_tasks: Tasks for exporting data in various formats
- asf.medical.tasks.ml_inference_tasks: Tasks for ML model inference

Usage:
    python -m asf.medical.run_workers [--processes N] [--threads M]

Options:
    --processes N    Number of worker processes (default: 2)
    --threads M      Number of worker threads per process (default: 8)
"""

import os
import sys
import logging
import argparse
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """Run Dramatiq workers for processing background tasks.

    This function parses command-line arguments to configure the number of worker
    processes and threads, then starts the Dramatiq workers to process tasks from
    the registered task modules.

    Args:
        None: Arguments are parsed from command line

    Returns:
        None: The function does not return as it runs the Dramatiq workers
        which block until terminated
    """
    parser = argparse.ArgumentParser(description="Run Dramatiq workers")
    parser.add_argument(
        "--processes", type=int, default=2, help="Number of worker processes"
    )
    parser.add_argument(
        "--threads", type=int, default=8, help="Number of worker threads per process"
    )
    args = parser.parse_args()

    logger.info(f"Starting Dramatiq workers with {args.processes} processes and {args.threads} threads per process")



    from dramatiq.cli import main as dramatiq_main

    sys.argv = [
        "dramatiq",
        "asf.medical.tasks.export_tasks",
        "asf.medical.tasks.ml_inference_tasks",
        "--processes", str(args.processes),
        "--threads", str(args.threads),
    ]

    dramatiq_main()

if __name__ == "__main__":
    main()
