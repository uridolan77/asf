"""
WebSocket connection manager for the BO backend.
"""

import asyncio
from typing import Dict, List, Set, Any, Optional
from fastapi import WebSocket
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class ConnectionManager:
    """
    WebSocket connection manager.

    This class manages WebSocket connections and provides methods for broadcasting
    messages to connected clients.
    """

    def __init__(self):
        """Initialize the connection manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.task_subscribers: Dict[str, Set[WebSocket]] = {}
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """
        Connect a WebSocket client.

        Args:
            websocket: WebSocket connection
            client_id: Client ID
        """
        await websocket.accept()
        async with self.lock:
            self.active_connections[client_id] = websocket
        logger.info(f"WebSocket client connected: {client_id}")

    async def disconnect(self, client_id: str) -> None:
        """
        Disconnect a WebSocket client.

        Args:
            client_id: Client ID
        """
        async with self.lock:
            if client_id in self.active_connections:
                del self.active_connections[client_id]

            # Remove from task subscribers
            for task_id in list(self.task_subscribers.keys()):
                if client_id in self.task_subscribers[task_id]:
                    self.task_subscribers[task_id].remove(client_id)
                if not self.task_subscribers[task_id]:
                    del self.task_subscribers[task_id]

        logger.info(f"WebSocket client disconnected: {client_id}")

    async def subscribe_to_task(self, client_id: str, task_id: str) -> None:
        """
        Subscribe a client to a task.

        Args:
            client_id: Client ID
            task_id: Task ID
        """
        async with self.lock:
            if task_id not in self.task_subscribers:
                self.task_subscribers[task_id] = set()
            self.task_subscribers[task_id].add(client_id)

        logger.info(f"Client {client_id} subscribed to task {task_id}")

    async def unsubscribe_from_task(self, client_id: str, task_id: str) -> None:
        """
        Unsubscribe a client from a task.

        Args:
            client_id: Client ID
            task_id: Task ID
        """
        async with self.lock:
            if task_id in self.task_subscribers and client_id in self.task_subscribers[task_id]:
                self.task_subscribers[task_id].remove(client_id)
                if not self.task_subscribers[task_id]:
                    del self.task_subscribers[task_id]

        logger.info(f"Client {client_id} unsubscribed from task {task_id}")

    async def broadcast_to_task_subscribers(self, task_id: str, message: Dict[str, Any]) -> None:
        """
        Broadcast a message to all subscribers of a task.

        Args:
            task_id: Task ID
            message: Message to broadcast
        """
        subscribers = set()
        async with self.lock:
            if task_id in self.task_subscribers:
                subscribers = self.task_subscribers[task_id]

        for client_id in subscribers:
            if client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_json(message)
                except Exception as e:
                    logger.error(f"Error sending message to client {client_id}: {str(e)}")

    async def broadcast_task_progress(self, task_id: str, progress: float, stage: str, metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Broadcast task progress to subscribers.

        Args:
            task_id: Task ID
            progress: Progress (0-1)
            stage: Current processing stage
            metrics: Additional metrics
        """
        message = {
            "type": "progress",
            "task_id": task_id,
            "progress": progress,
            "stage": stage,
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat()
        }

        await self.broadcast_to_task_subscribers(task_id, message)

    async def broadcast_task_completed(self, task_id: str, result: Dict[str, Any]) -> None:
        """
        Broadcast task completion to subscribers.

        Args:
            task_id: Task ID
            result: Task result
        """
        message = {
            "type": "completed",
            "task_id": task_id,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }

        await self.broadcast_to_task_subscribers(task_id, message)

    async def broadcast_task_failed(self, task_id: str, error: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Broadcast task failure to subscribers.

        Args:
            task_id: Task ID
            error: Error message
            details: Additional error details
        """
        # Log the error for debugging
        logger.error(f"Task {task_id} failed: {error}")
        if details:
            logger.error(f"Error details: {json.dumps(details, indent=2)}")

        message = {
            "type": "failed",
            "task_id": task_id,
            "error": error,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }

        await self.broadcast_to_task_subscribers(task_id, message)

# Create a global connection manager
connection_manager = ConnectionManager()
