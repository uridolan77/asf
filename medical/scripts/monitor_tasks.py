Script to monitor tasks in the Medical Research Synthesizer.
This script provides a command-line interface for monitoring tasks.
import sys
import asyncio
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from asf.medical.core.logging_config import get_logger
from asf.medical.storage.database import get_db_session
from asf.medical.storage.repositories.task_repository import TaskRepository
from asf.medical.storage.models.task import TaskStatus
logger = get_logger(__name__)
class TaskMonitor:
    Task monitor class.
        async with get_db_session() as db:
            return await self.task_repository.get_task_count_by_status(db)
    async def get_pending_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get pending tasks.
        Args:
            limit: Maximum number of tasks to return
        Returns:
            List of pending tasks
        """
        async with get_db_session() as db:
            tasks = await self.task_repository.get_pending_tasks_for_processing(db, limit)
            return [task.to_dict() for task in tasks]
    async def get_tasks_by_status(self, status: TaskStatus, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get tasks by status.
        Args:
            status: Task status
            limit: Maximum number of tasks to return
        Returns:
            List of tasks
        """
        async with get_db_session() as db:
            tasks = await self.task_repository.get_tasks_by_status(db, status, limit)
            return [task.to_dict() for task in tasks]
    async def get_task_with_events(self, task_id: str) -> Dict[str, Any]:
        """
        Get a task with its events.
        Args:
            task_id: Task ID
        Returns:
            Task with events
        """
        async with get_db_session() as db:
            task, events = await self.task_repository.get_task_with_events(db, task_id)
            if not task:
                return {"error": f"Task not found: {task_id}"}
            return {
                "task": task.to_dict(),
                "events": [event.to_dict() for event in events]
            }
    async def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """
        Cancel a task.
        Args:
            task_id: Task ID
        Returns:
            Result of cancellation
        """
        async with get_db_session() as db:
            task = await self.task_repository.cancel_task(db, task_id)
            if not task:
                return {"error": f"Task not found or could not be cancelled: {task_id}"}
            return {
                "message": f"Task {task_id} cancelled successfully",
                "task": task.to_dict()
            }
    async def cleanup_old_tasks(self, days: int = 30) -> Dict[str, Any]:
        """
        Clean up old tasks.
        Args:
            days: Age in days for tasks to be considered old
        Returns:
            Result of cleanup
        """
        async with get_db_session() as db:
            tasks_deleted = await self.task_repository.cleanup_old_tasks(db, days)
            messages_deleted = await self.task_repository.cleanup_old_dead_letters(db, days)
            return {
                "tasks_deleted": tasks_deleted,
                "messages_deleted": messages_deleted,
                "message": f"Cleaned up {tasks_deleted} tasks and {messages_deleted} dead letter messages older than {days} days"
            }
async def show_task_counts():
    """Show task counts by status."""
    monitor = TaskMonitor()
    counts = await monitor.get_task_counts()
    print("\nTask Counts by Status:")
    print("=====================")
    for status, count in counts.items():
        print(f"{status}: {count}")
    print()
async def show_pending_tasks(limit: int = 10):
    """
    Show pending tasks.
    Args:
        limit: Maximum number of tasks to show
    """
    monitor = TaskMonitor()
    tasks = await monitor.get_pending_tasks(limit)
    print(f"\nPending Tasks (limit: {limit}):")
    print("=======================" + "=" * len(str(limit)))
    if not tasks:
        print("No pending tasks found.")
        return
    for task in tasks:
        print(f"ID: {task['id']}")
        print(f"Type: {task['type']}")
        print(f"Status: {task['status']}")
        print(f"Progress: {task['progress']}%")
        print(f"Created: {task['created_at']}")
        print(f"Updated: {task['updated_at']}")
        print(f"Retry Count: {task['retry_count']}/{task['max_retries']}")
        print("-" * 50)
    print()
async def show_tasks_by_status(status: str, limit: int = 10):
    """
    Show tasks by status.
    Args:
        status: Task status
        limit: Maximum number of tasks to show
    """
    try:
        task_status = TaskStatus(status)
    except ValueError:
        print(f"Invalid task status: {status}")
        print(f"Valid statuses: {', '.join([s.value for s in TaskStatus])}")
        return
    monitor = TaskMonitor()
    tasks = await monitor.get_tasks_by_status(task_status, limit)
    print(f"\nTasks with Status '{status}' (limit: {limit}):")
    print("=======================" + "=" * len(status) + "=" * len(str(limit)))
    if not tasks:
        print(f"No tasks with status '{status}' found.")
        return
    for task in tasks:
        print(f"ID: {task['id']}")
        print(f"Type: {task['type']}")
        print(f"Progress: {task['progress']}%")
        print(f"Created: {task['created_at']}")
        print(f"Updated: {task['updated_at']}")
        print(f"Message: {task['message']}")
        print("-" * 50)
    print()
async def show_task_details(task_id: str):
    """
    Show task details.
    Args:
        task_id: Task ID
    """
    monitor = TaskMonitor()
    result = await monitor.get_task_with_events(task_id)
    if "error" in result:
        print(f"\nError: {result['error']}")
        return
    task = result["task"]
    events = result["events"]
    print(f"\nTask Details for {task_id}:")
    print("=======================" + "=" * len(task_id))
    print(f"Type: {task['type']}")
    print(f"Status: {task['status']}")
    print(f"Progress: {task['progress']}%")
    print(f"User ID: {task['user_id']}")
    print(f"Created: {task['created_at']}")
    print(f"Updated: {task['updated_at']}")
    print(f"Started: {task['started_at']}")
    print(f"Completed: {task['completed_at']}")
    print(f"Retry Count: {task['retry_count']}/{task['max_retries']}")
    print(f"Next Retry: {task['next_retry_at']}")
    print(f"Worker ID: {task['worker_id']}")
    print(f"Cancellable: {task['cancellable']}")
    print(f"Cancelled: {task['cancelled']}")
    print(f"Message: {task['message']}")
    if task['error']:
        print(f"\nError:")
        print(task['error'])
    if task['result']:
        print(f"\nResult:")
        print(json.dumps(task['result'], indent=2))
    if task['params']:
        print(f"\nParameters:")
        print(json.dumps(task['params'], indent=2))
    print(f"\nEvents ({len(events)}):")
    print("==========")
    for event in events:
        print(f"Type: {event['event_type']}")
        print(f"Time: {event['created_at']}")
        if event['event_data']:
            print(f"Data: {json.dumps(event['event_data'], indent=2)}")
        print("-" * 50)
    print()
async def cancel_task(task_id: str):
    """
    Cancel a task.
    Args:
        task_id: Task ID
    """
    monitor = TaskMonitor()
    result = await monitor.cancel_task(task_id)
    if "error" in result:
        print(f"\nError: {result['error']}")
        return
    print(f"\n{result['message']}")
    print()
async def cleanup_tasks(days: int = 30):
    """
    Clean up old tasks.
    Args:
        days: Age in days for tasks to be considered old
    """
    monitor = TaskMonitor()
    result = await monitor.cleanup_old_tasks(days)
    print(f"\n{result['message']}")
    print()
async def interactive_mode():
    """Run the monitor in interactive mode."""
    while True:
        print("\nTask Monitor - Interactive Mode")
        print("==============================")
        print("1. Show task counts")
        print("2. Show pending tasks")
        print("3. Show tasks by status")
        print("4. Show task details")
        print("5. Cancel task")
        print("6. Clean up old tasks")
        print("0. Exit")
        choice = input("\nEnter your choice: ")
        if choice == "0":
            break
        elif choice == "1":
            await show_task_counts()
        elif choice == "2":
            limit = int(input("Enter limit (default: 10): ") or "10")
            await show_pending_tasks(limit)
        elif choice == "3":
            status = input("Enter status: ")
            limit = int(input("Enter limit (default: 10): ") or "10")
            await show_tasks_by_status(status, limit)
        elif choice == "4":
            task_id = input("Enter task ID: ")
            await show_task_details(task_id)
        elif choice == "5":
            task_id = input("Enter task ID: ")
            confirm = input(f"Are you sure you want to cancel task {task_id}? (y/n): ")
            if confirm.lower() == "y":
                await cancel_task(task_id)
        elif choice == "6":
            days = int(input("Enter age in days (default: 30): ") or "30")
            confirm = input(f"Are you sure you want to clean up tasks older than {days} days? (y/n): ")
            if confirm.lower() == "y":
                await cleanup_tasks(days)
        else:
            print("Invalid choice. Please try again.")
def main():
    Main entry point.
    parser = argparse.ArgumentParser(description="Task Monitor")
    # Add subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    # Counts command
    counts_parser = subparsers.add_parser("counts", help="Show task counts by status")
    # Pending command
    pending_parser = subparsers.add_parser("pending", help="Show pending tasks")
    pending_parser.add_argument("--limit", type=int, default=10, help="Maximum number of tasks to show")
    # Status command
    status_parser = subparsers.add_parser("status", help="Show tasks by status")
    status_parser.add_argument("status", help="Task status")
    status_parser.add_argument("--limit", type=int, default=10, help="Maximum number of tasks to show")
    # Details command
    details_parser = subparsers.add_parser("details", help="Show task details")
    details_parser.add_argument("task_id", help="Task ID")
    # Cancel command
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a task")
    cancel_parser.add_argument("task_id", help="Task ID")
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old tasks")
    cleanup_parser.add_argument("--days", type=int, default=30, help="Age in days for tasks to be considered old")
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Run in interactive mode")
    # Parse arguments
    args = parser.parse_args()
    # Run the appropriate command
    if args.command == "counts":
        asyncio.run(show_task_counts())
    elif args.command == "pending":
        asyncio.run(show_pending_tasks(args.limit))
    elif args.command == "status":
        asyncio.run(show_tasks_by_status(args.status, args.limit))
    elif args.command == "details":
        asyncio.run(show_task_details(args.task_id))
    elif args.command == "cancel":
        asyncio.run(cancel_task(args.task_id))
    elif args.command == "cleanup":
        asyncio.run(cleanup_tasks(args.days))
    elif args.command == "interactive":
        asyncio.run(interactive_mode())
    else:
        parser.print_help()
if __name__ == "__main__":
    main()