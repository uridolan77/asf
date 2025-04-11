Script to retry failed tasks in the Medical Research Synthesizer.

This script is intended to be run as a scheduled job to retry failed tasks.

import sys
import asyncio
import argparse
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from asf.medical.core.logging_config import get_logger
from asf.medical.storage.database import get_db_session
from asf.medical.storage.repositories.task_repository import TaskRepository
from asf.medical.storage.models.task import TaskStatus
from asf.medical.core.messaging.producer import get_message_producer

logger = get_logger(__name__)

async def retry_failed_tasks(max_age_days: int = 7, max_retries: int = 3):
    """
    Retry failed tasks.
    
    Args:
        max_age_days: Maximum age in days for tasks to be considered for retry
        max_retries: Maximum number of retries
    """
    logger.info(f"Starting task retry (max age: {max_age_days} days, max retries: {max_retries})")
    
    task_repository = TaskRepository()
    producer = get_message_producer()
    
    # Calculate the cutoff date
    cutoff_date = datetime.now(datetime.timezone.utc) - timedelta(days=max_age_days)
    
    async with get_db_session() as db:
        # Get failed tasks
        tasks = await task_repository.get_tasks_by_status(db, TaskStatus.FAILED, limit=100)
        
        # Filter tasks by age and retry count
        eligible_tasks = [
            task for task in tasks
            if task.created_at >= cutoff_date and task.retry_count < max_retries
        ]
        
        logger.info(f"Found {len(eligible_tasks)} eligible failed tasks to retry")
        
        # Retry each task
        retried_count = 0
        for task in eligible_tasks:
            try:
                # Mark the task for retry
                await task_repository.mark_task_for_retry(
                    db=db,
                    task_id=task.id,
                    error=f"Scheduled retry by retry_failed_tasks.py",
                    retry_delay=60  # 1 minute delay
                )
                
                # Republish the task
                await producer.publish_task(
                    task_type=task.type,
                    task_data=task.params or {},
                    task_id=task.id
                )
                
                retried_count += 1
                logger.info(f"Retried task {task.id} (type: {task.type})")
            except Exception as e:
                logger.error(f"Error retrying task {task.id}: {str(e)}", exc_info=e)
    
    logger.info(f"Task retry completed: {retried_count} tasks retried")
    
    return retried_count

def main():
    Main entry point.
    parser = argparse.ArgumentParser(description="Task Retry")
    parser.add_argument("--max-age-days", type=int, default=7, help="Maximum age in days for tasks to be considered for retry")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries")
    args = parser.parse_args()
    
    retried_count = asyncio.run(retry_failed_tasks(args.max_age_days, args.max_retries))
    
    print(f"Retried {retried_count} failed tasks")

if __name__ == "__main__":
    main()
