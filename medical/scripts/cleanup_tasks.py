Task Cleanup Script

This script cleans up old tasks from the persistent storage.

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asf.medical.core.persistent_task_storage import task_storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """Clean up old tasks from the persistent storage.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    parser = argparse.ArgumentParser(description="Clean up old tasks from the persistent storage")
    parser.add_argument(
        "--days", type=int, default=7, help="Age in days of tasks to clean up (default: 7)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Dry run (don't actually delete tasks)"
    )
    args = parser.parse_args()
    
    max_age = args.days * 24 * 60 * 60
    
    logger.info(f"Cleaning up tasks older than {args.days} days ({max_age} seconds)")
    
    if args.dry_run:
        logger.info("Dry run mode - tasks will not be deleted")
        
        tasks = task_storage.list_tasks(limit=0)  # No limit
        
        old_tasks = 0
        current_time = datetime.now().timestamp()
        
        for task in tasks:
            updated_at = task.get("updated_at", 0)
            age = current_time - updated_at
            
            if age > max_age:
                old_tasks += 1
                logger.info(f"Would delete task {task.get('task_id')} (age: {timedelta(seconds=age)})")
        
        logger.info(f"Would delete {old_tasks} tasks out of {len(tasks)} total tasks")
    else:
        deleted_count = task_storage.cleanup_old_tasks(max_age)
        logger.info(f"Deleted {deleted_count} old tasks")

if __name__ == "__main__":
    main()
