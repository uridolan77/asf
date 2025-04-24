"""
WebSocket endpoints for task updates.
This module provides WebSocket endpoints for real-time task updates.
"""
import json
import asyncio
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from asf.medical.core.logging_config import get_logger
from asf.medical.storage.database import get_db_session
from asf.medical.storage.models.task import Task, TaskEvent
from asf.medical.api.dependencies import get_current_user_ws
from asf.medical.storage.models import MedicalUser

from asf.medical.core.exceptions import OperationError, ValidationError

logger = get_logger(__name__)
class TaskUpdateManager:
    """
    Manager for task update WebSocket connections.
    This class manages WebSocket connections for task updates,
    allowing clients to subscribe to updates for specific tasks.
    """
    def __init__(self):
        """
        Initialize the task update manager.
        
        Args:
        
        Connect a WebSocket client.
        Args:
            websocket: WebSocket connection
            user: User (optional)
        """
        await websocket.accept()
        async with self.lock:
            client_id = id(websocket)
            self.active_connections[client_id] = websocket
            if user:
                if user.id not in self.user_connections:
                    self.user_connections[user.id] = set()
                self.user_connections[user.id].add(websocket)
        logger.debug(f"WebSocket client connected: {client_id}")
    async def disconnect(self, websocket: WebSocket, user: Optional[MedicalUser] = None) -> None:
        """
        Disconnect a WebSocket client.
        Args:
            websocket: WebSocket connection
            user: User (optional)
        """
        async with self.lock:
            client_id = id(websocket)
            # Remove from active connections
            if client_id in self.active_connections:
                del self.active_connections[client_id]
            # Remove from user connections
            if user and user.id in self.user_connections:
                self.user_connections[user.id].discard(websocket)
                if not self.user_connections[user.id]:
                    del self.user_connections[user.id]
            # Remove from task subscribers
            for task_id, subscribers in list(self.task_subscribers.items()):
                subscribers.discard(websocket)
                if not subscribers:
                    del self.task_subscribers[task_id]
            # Remove from all subscribers
            self.all_subscribers.discard(websocket)
        logger.debug(f"WebSocket client disconnected: {client_id}")
    async def subscribe_to_task(self, websocket: WebSocket, task_id: str) -> None:
        """
        Subscribe to updates for a specific task.
        Args:
            websocket: WebSocket connection
            task_id: Task ID
        """
        async with self.lock:
            if task_id not in self.task_subscribers:
                self.task_subscribers[task_id] = set()
            self.task_subscribers[task_id].add(websocket)
        logger.debug(f"Client subscribed to task {task_id}")
        # Send confirmation
        await websocket.send_json({
            "type": "subscription",
            "task_id": task_id,
            "status": "subscribed",
            "timestamp": datetime.now(datetime.timezone.utc).isoformat()
        })
    async def unsubscribe_from_task(self, websocket: WebSocket, task_id: str) -> None:
        """
        Unsubscribe from updates for a specific task.
        Args:
            websocket: WebSocket connection
            task_id: Task ID
        """
        async with self.lock:
            if task_id in self.task_subscribers:
                self.task_subscribers[task_id].discard(websocket)
                if not self.task_subscribers[task_id]:
                    del self.task_subscribers[task_id]
        logger.debug(f"Client unsubscribed from task {task_id}")
        # Send confirmation
        await websocket.send_json({
            "type": "subscription",
            "task_id": task_id,
            "status": "unsubscribed",
            "timestamp": datetime.now(datetime.timezone.utc).isoformat()
        })
    async def subscribe_to_all_tasks(self, websocket: WebSocket) -> None:
        """
        Subscribe to updates for all tasks.
        Args:
            websocket: WebSocket connection
        """
        async with self.lock:
            self.all_subscribers.add(websocket)
        logger.debug(f"Client subscribed to all tasks")
        # Send confirmation
        await websocket.send_json({
            "type": "subscription",
            "status": "subscribed_all",
            "timestamp": datetime.now(datetime.timezone.utc).isoformat()
        })
    async def unsubscribe_from_all_tasks(self, websocket: WebSocket) -> None:
        """
        Unsubscribe from updates for all tasks.
        Args:
            websocket: WebSocket connection
        """
        async with self.lock:
            self.all_subscribers.discard(websocket)
        logger.debug(f"Client unsubscribed from all tasks")
        # Send confirmation
        await websocket.send_json({
            "type": "subscription",
            "status": "unsubscribed_all",
            "timestamp": datetime.now(datetime.timezone.utc).isoformat()
        })
    async def broadcast_task_update(self, task: Task, event_type: str, data: Dict[str, Any]) -> None:
        """
        Broadcast a task update to subscribers.
        Args:
            task: Task
            event_type: Event type
            data: Event data
        """
        message = {
            "type": "task_update",
            "task_id": task.id,
            "task_type": task.type,
            "status": task.status.value if task.status else None,
            "progress": task.progress,
            "message": task.message,
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now(datetime.timezone.utc).isoformat()
        }
        # Get subscribers for this task
        subscribers = set()
        async with self.lock:
            # Add task-specific subscribers
            if task.id in self.task_subscribers:
                subscribers.update(self.task_subscribers[task.id])
            # Add user-specific subscribers
            if task.user_id and task.user_id in self.user_connections:
                subscribers.update(self.user_connections[task.user_id])
            # Add all-task subscribers
            subscribers.update(self.all_subscribers)
        # Send the update to all subscribers
        for websocket in subscribers:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending task update to WebSocket client: {str(e)}", exc_info=e)
                raise OperationError(f"Operation failed: {str(e)}")
                # Don't disconnect here, as it could be a temporary issue
    async def broadcast_task_created(self, task: Task) -> None:
        """
        Broadcast a task created event.
        Args:
            task: Task
        """
        await self.broadcast_task_update(
            task=task,
            event_type="created",
            data={
                "created_at": task.created_at.isoformat() if task.created_at else None
            }
        )
    async def broadcast_task_status_changed(self, task: Task, previous_status: str) -> None:
        """
        Broadcast a task status changed event.
        Args:
            task: Task
            previous_status: Previous status
        """
        await self.broadcast_task_update(
            task=task,
            event_type="status_changed",
            data={
                "previous_status": previous_status,
                "new_status": task.status.value if task.status else None
            }
        )
    async def broadcast_task_progress(self, task: Task) -> None:
        """
        Broadcast a task progress event.
        Args:
            task: Task
        """
        await self.broadcast_task_update(
            task=task,
            event_type="progress",
            data={
                "progress": task.progress,
                "message": task.message
            }
        )
    async def broadcast_task_completed(self, task: Task) -> None:
        """
        Broadcast a task completed event.
        Args:
            task: Task
        """
        await self.broadcast_task_update(
            task=task,
            event_type="completed",
            data={
                "result": task.result,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None
            }
        )
    async def broadcast_task_failed(self, task: Task) -> None:
        """
        Broadcast a task failed event.
        Args:
            task: Task
        """
        await self.broadcast_task_update(
            task=task,
            event_type="failed",
            data={
                "error": task.error,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None
            }
        )
    async def broadcast_task_cancelled(self, task: Task) -> None:
        """
        Broadcast a task cancelled event.
        Args:
            task: Task
        """
        await self.broadcast_task_update(
            task=task,
            event_type="cancelled",
            data={
                "completed_at": task.completed_at.isoformat() if task.completed_at else None
            }
        )
    async def broadcast_task_event(self, task: Task, event: TaskEvent) -> None:
        """
        Broadcast a task event.
        Args:
            task: Task
            event: Task event
        """
        await self.broadcast_task_update(
            task=task,
            event_type=event.event_type,
            data=event.event_data or {}
        )
# Create a global task update manager
task_update_manager = TaskUpdateManager()
async def handle_task_updates(
    websocket: WebSocket,
    db: AsyncSession = Depends(get_db_session),
    user: Optional[MedicalUser] = Depends(get_current_user_ws)
):
    """
    Handle WebSocket connections for task updates.
    Args:
        websocket: WebSocket connection
        db: Database session
        user: Current user (optional)
    """
    await task_update_manager.connect(websocket, user)
    try:
        while True:
            # Wait for messages from the client
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                # Handle different message types
                if message.get("type") == "subscribe":
                    task_id = message.get("task_id")
                    if task_id:
                        # Subscribe to a specific task
                        await task_update_manager.subscribe_to_task(websocket, task_id)
                    else:
                        # Subscribe to all tasks
                        await task_update_manager.subscribe_to_all_tasks(websocket)
                elif message.get("type") == "unsubscribe":
                    task_id = message.get("task_id")
                    if task_id:
                        # Unsubscribe from a specific task
                        await task_update_manager.unsubscribe_from_task(websocket, task_id)
                    else:
                        # Unsubscribe from all tasks
                        await task_update_manager.unsubscribe_from_all_tasks(websocket)
                elif message.get("type") == "ping":
                    # Respond to ping
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now(datetime.timezone.utc).isoformat()
                    })
                else:
                    # Unknown message type
                    await websocket.send_json({
                        "type": "error",
                        "message": "Unknown message type",
                        "timestamp": datetime.now(datetime.timezone.utc).isoformat()
                    })
            except json.JSONDecodeError:
                logger.error(f"Error: {str(e)}")
                # Invalid JSON
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON",
                    "timestamp": datetime.now(datetime.timezone.utc).isoformat()
                })
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                # Other errors
                raise ValidationError(f"Validation failed: {str(e)}")
                logger.error(f"Error handling WebSocket message: {str(e)}", exc_info=e)
                await websocket.send_json({
                    "type": "error",
                    "message": "Internal server error",
                    "timestamp": datetime.now(datetime.timezone.utc).isoformat()
                })
    except WebSocketDisconnect:
        logger.error(f"Error: {str(e)}")
        # Client disconnected
        await task_update_manager.disconnect(websocket, user)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        # Other errors
        raise ValidationError(f"Validation failed: {str(e)}")
        logger.error(f"Error in WebSocket connection: {str(e)}", exc_info=e)
        await task_update_manager.disconnect(websocket, user)