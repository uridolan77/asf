"""
Medical Research Synthesizer Application Launcher.

This script initializes and starts all components of the Medical Research Synthesizer
application, including the API server, task scheduler, and knowledge base updater.
It provides command-line options to selectively start specific components or
initialize the database.

Usage:
    python -m asf.medical.scripts.run_app [options]

Options:
    --init-db           Initialize the database before starting components
    --api-only          Run only the API server
    --scheduler-only    Run only the task scheduler
    --kb-updater-only   Run only the knowledge base updater

If no component-specific options are provided, the script starts all components.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from asf.medical.core.logging import setup_logging
from asf.medical.storage.database import init_db
from asf.medical.orchestration import RayManager, TaskScheduler, KnowledgeBaseUpdater

logger = setup_logging()

def main():
    """Run the Medical Research Synthesizer application.

    This function parses command-line arguments and starts the appropriate components
    of the application based on the provided options. It can:
    1. Initialize the database with required tables and initial data
    2. Start the API server to handle HTTP requests
    3. Start the task scheduler to manage background tasks
    4. Start the knowledge base updater to keep knowledge bases current

    The function blocks until interrupted with Ctrl+C when running the scheduler
    or knowledge base updater. When running the API, it blocks until the server
    is stopped.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Run the Medical Research Synthesizer application")
    parser.add_argument("--init-db", action="store_true", help="Initialize the database")
    parser.add_argument("--api-only", action="store_true", help="Run only the API")
    parser.add_argument("--scheduler-only", action="store_true", help="Run only the task scheduler")
    parser.add_argument("--kb-updater-only", action="store_true", help="Run only the knowledge base updater")
    args = parser.parse_args()

    logger.info("Starting Medical Research Synthesizer application")

    if args.init_db:
        logger.info("Initializing database")
        init_db()

        from asf.medical.scripts.init_db import init
        init()

    if args.api_only or (not args.scheduler_only and not args.kb_updater_only):
        logger.info("Starting API")

        from asf.medical.scripts.run_api import main as run_api
        run_api()

    if args.scheduler_only:
        logger.info("Starting task scheduler")

        ray_manager = RayManager()
        ray_manager.initialize()

        task_scheduler = TaskScheduler()
        task_scheduler.start()

        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping task scheduler")
            task_scheduler.stop()

    if args.kb_updater_only:
        logger.info("Starting knowledge base updater")

        kb_updater = KnowledgeBaseUpdater()
        kb_updater.start()

        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping knowledge base updater")
            kb_updater.stop()

if __name__ == "__main__":
    main()
