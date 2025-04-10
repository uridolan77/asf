"""
Script to run the full Medical Research Synthesizer application.

This script initializes and starts all components of the application.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from asf.medical.core.config import settings
from asf.medical.core.logging import setup_logging
from asf.medical.storage.database import init_db
from asf.medical.orchestration import RayManager, TaskScheduler, KnowledgeBaseUpdater

# Set up logging
logger = setup_logging()

def main():
    """Run the full application."""
    parser = argparse.ArgumentParser(description="Run the Medical Research Synthesizer application")
    parser.add_argument("--init-db", action="store_true", help="Initialize the database")
    parser.add_argument("--api-only", action="store_true", help="Run only the API")
    parser.add_argument("--scheduler-only", action="store_true", help="Run only the task scheduler")
    parser.add_argument("--kb-updater-only", action="store_true", help="Run only the knowledge base updater")
    args = parser.parse_args()
    
    logger.info("Starting Medical Research Synthesizer application")
    
    # Initialize database if requested
    if args.init_db:
        logger.info("Initializing database")
        init_db()
        
        # Import and run the init_db script
        from asf.medical.scripts.init_db import init
        init()
    
    # Run API if requested or if no specific component is requested
    if args.api_only or (not args.scheduler_only and not args.kb_updater_only):
        logger.info("Starting API")
        
        # Import and run the API
        from asf.medical.scripts.run_api import main as run_api
        run_api()
    
    # Run task scheduler if requested
    if args.scheduler_only:
        logger.info("Starting task scheduler")
        
        # Initialize Ray
        ray_manager = RayManager()
        ray_manager.initialize()
        
        # Start task scheduler
        task_scheduler = TaskScheduler()
        task_scheduler.start()
        
        # Keep the script running
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping task scheduler")
            task_scheduler.stop()
    
    # Run knowledge base updater if requested
    if args.kb_updater_only:
        logger.info("Starting knowledge base updater")
        
        # Start knowledge base updater
        kb_updater = KnowledgeBaseUpdater()
        kb_updater.start()
        
        # Keep the script running
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping knowledge base updater")
            kb_updater.stop()

if __name__ == "__main__":
    main()
