"""
Run Dramatiq Workers

This script starts Dramatiq workers for processing background tasks.
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
    """Run Dramatiq workers.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
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
