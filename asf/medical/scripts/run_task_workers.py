"""
Task Workers Launcher for the Medical Research Synthesizer.

This script initializes and starts task workers that consume messages from the RabbitMQ
message broker. The workers process various types of tasks including search tasks,
analysis tasks, and export tasks. Each task type is handled by a specific handler
that implements the business logic for that task.

The script sets up signal handlers to gracefully shutdown the workers when the
process receives termination signals (SIGINT, SIGTERM).

Usage:
    python -m asf.medical.scripts.run_task_workers [--debug]

Options:
    --debug    Enable debug logging for more detailed output

Requirements:
    - RabbitMQ must be enabled in the application settings (RABBITMQ_ENABLED=true)
"""

import sys
import asyncio
import argparse
import signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from asf.medical.core.logging_config import get_logger
from asf.medical.core.config import settings
from asf.medical.core.messaging.initialization import initialize_messaging_system, shutdown_messaging_system
from asf.medical.core.messaging.consumer import get_message_consumer
from asf.medical.core.messaging.schemas import TaskType
from asf.medical.services.task_handlers import SearchTaskHandler, AnalysisTaskHandler, ExportTaskHandler

logger = get_logger(__name__)

async def run_workers():
    """
    Initialize and run task workers for processing messages from the queue.

    This function performs the following operations:
    1. Initializes the messaging system (RabbitMQ connection)
    2. Creates a message consumer
    3. Registers task handlers for different task types
    4. Starts consuming messages from the queue
    5. Keeps the workers running until interrupted

    The function handles graceful shutdown when interrupted, ensuring that
    the messaging system is properly closed.

    Returns:
        None

    Raises:
        asyncio.CancelledError: When the task is cancelled during shutdown
    """
    # Initialize the messaging system
    await initialize_messaging_system()

    # Get the message consumer
    consumer = get_message_consumer()

    # Register task handlers
    search_handler = SearchTaskHandler()
    analysis_handler = AnalysisTaskHandler()
    export_handler = ExportTaskHandler()

    # Register search task handlers
    consumer.register_task_handler(TaskType.SEARCH_PUBMED, search_handler)
    consumer.register_task_handler(TaskType.SEARCH_CLINICAL_TRIALS, search_handler)
    consumer.register_task_handler(TaskType.SEARCH_KNOWLEDGE_BASE, search_handler)

    # Register analysis task handlers
    consumer.register_task_handler(TaskType.ANALYZE_CONTRADICTIONS, analysis_handler)
    consumer.register_task_handler(TaskType.ANALYZE_BIAS, analysis_handler)
    consumer.register_task_handler(TaskType.ANALYZE_TRENDS, analysis_handler)

    # Register export task handlers
    consumer.register_task_handler(TaskType.EXPORT_RESULTS, export_handler)
    consumer.register_task_handler(TaskType.EXPORT_ANALYSIS, export_handler)

    # Start consuming messages
    await consumer.start()

    logger.info("Task workers started")

    # Keep the workers running until interrupted
    try:
        # Create a future that will never complete
        future = asyncio.Future()
        await future
    except asyncio.CancelledError:
        logger.info("Task workers interrupted")
    finally:
        # Shutdown the messaging system
        await shutdown_messaging_system()
        logger.info("Task workers stopped")

def main():
    """
    Main entry point for the task workers script.

    This function performs the following operations:
    1. Parses command-line arguments
    2. Configures logging based on the debug flag
    3. Verifies that RabbitMQ is enabled in the settings
    4. Sets up signal handlers for graceful shutdown
    5. Runs the task workers in the event loop

    The function handles keyboard interrupts and ensures proper cleanup
    of resources when the script is terminated.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Run task workers")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        import logging
        logging.getLogger("asf.medical").setLevel(logging.DEBUG)

    if not settings.RABBITMQ_ENABLED:
        logger.error("RabbitMQ messaging is disabled. Set RABBITMQ_ENABLED=true to enable it.")
        sys.exit(1)

    # Set up signal handlers
    loop = asyncio.get_event_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))

    # Run the workers
    try:
        loop.run_until_complete(run_workers())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        loop.close()

async def shutdown():
    """
    Shutdown the task workers gracefully.

    This function is called when the process receives a termination signal (SIGINT, SIGTERM).
    It performs the following operations:
    1. Logs the shutdown initiation
    2. Identifies all running asyncio tasks except the current one
    3. Cancels all identified tasks
    4. Waits for all tasks to complete or raise exceptions
    5. Stops the event loop

    This ensures that all resources are properly released and ongoing operations
    are completed or cancelled before the process exits.

    Returns:
        None
    """
    logger.info("Shutting down task workers...")

    # Get all running tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    # Cancel all running tasks
    for task in tasks:
        task.cancel()

    # Wait for all tasks to complete
    await asyncio.gather(*tasks, return_exceptions=True)

    # Stop the event loop
    asyncio.get_event_loop().stop()

if __name__ == "__main__":
    main()
