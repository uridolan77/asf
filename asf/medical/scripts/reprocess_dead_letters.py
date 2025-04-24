Script to reprocess dead letter messages in the Medical Research Synthesizer.

This script is intended to be run as a scheduled job to reprocess dead letter messages.

import sys
import asyncio
import argparse
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from asf.medical.core.logging_config import get_logger
from asf.medical.storage.database import get_db_session
from asf.medical.storage.repositories.task_repository import TaskRepository
from asf.medical.core.messaging.producer import get_message_producer

logger = get_logger(__name__)

async def reprocess_dead_letters(max_age_days: int = 7, max_retries: int = 3):
    """
    Reprocess dead letter messages.
    
    Args:
        max_age_days: Maximum age in days for messages to be considered for reprocessing
        max_retries: Maximum number of retries
    """
    logger.info(f"Starting dead letter reprocessing (max age: {max_age_days} days, max retries: {max_retries})")
    
    task_repository = TaskRepository()
    producer = get_message_producer()
    
    # Calculate the cutoff date
    cutoff_date = datetime.now(datetime.timezone.utc) - timedelta(days=max_age_days)
    
    async with get_db_session() as db:
        # Get dead letter messages that haven't been reprocessed
        messages = await task_repository.get_dead_letter_messages(db, limit=100, reprocessed=False)
        
        # Filter messages by age and retry count
        eligible_messages = [
            message for message in messages
            if message.created_at >= cutoff_date and message.retry_count < max_retries
        ]
        
        logger.info(f"Found {len(eligible_messages)} eligible dead letter messages to reprocess")
        
        # Reprocess each message
        reprocessed_count = 0
        for message in eligible_messages:
            try:
                # Republish the message
                await producer.publish_raw_message(
                    exchange=message.exchange,
                    routing_key=message.routing_key,
                    message=message.message,
                    headers=message.headers
                )
                
                # Mark the message as reprocessed
                await task_repository.mark_dead_letter_as_reprocessed(db, message.id)
                
                reprocessed_count += 1
                logger.info(f"Reprocessed dead letter message {message.id}")
            except Exception as e:
                logger.error(f"Error reprocessing dead letter message {message.id}: {str(e)}", exc_info=e)
    
    logger.info(f"Dead letter reprocessing completed: {reprocessed_count} messages reprocessed")
    
    return reprocessed_count

def main():
    Main entry point.
    parser = argparse.ArgumentParser(description="Dead Letter Reprocessing")
    parser.add_argument("--max-age-days", type=int, default=7, help="Maximum age in days for messages to be considered for reprocessing")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries")
    args = parser.parse_args()
    
    reprocessed_count = asyncio.run(reprocess_dead_letters(args.max_age_days, args.max_retries))
    
    print(f"Reprocessed {reprocessed_count} dead letter messages")

if __name__ == "__main__":
    main()
