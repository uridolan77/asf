"""
Enhanced WebSocket connection manager for MCP provider.

This module provides a robust connection manager for MCP WebSocket connections
with features like message queuing for offline periods, automatic reconnection,
and improved error handling.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Set, Any, Optional, Callable, Awaitable, Tuple

from fastapi import WebSocket
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)

# Message queue settings
MAX_QUEUE_SIZE = 100  # Maximum number of messages to queue per client
MAX_QUEUE_AGE_SECONDS = 300  # Maximum age of queued messages (5 minutes)
QUEUE_CLEANUP_INTERVAL = 60  # Cleanup interval for message queues (1 minute)

# Heartbeat settings
HEARTBEAT_INTERVAL = 30  # Send heartbeat every 30 seconds
HEARTBEAT_TIMEOUT = 45  # Consider connection dead after 45 seconds without response


class QueuedMessage:
    """
    Represents a message in the queue.
    """
    
    def __init__(self, message: Dict[str, Any], priority: int = 0):
        """
        Initialize a queued message.
        
        Args:
            message: The message to queue
            priority: Message priority (higher = more important)
        """
        self.message = message
        self.timestamp = datetime.utcnow()
        self.priority = priority
        self.attempts = 0
    
    def is_expired(self) -> bool:
        """
        Check if the message has expired.
        
        Returns:
            True if the message has expired, False otherwise
        """
        age = (datetime.utcnow() - self.timestamp).total_seconds()
        return age > MAX_QUEUE_AGE_SECONDS
    
    def increment_attempts(self) -> None:
        """Increment the number of delivery attempts."""
        self.attempts += 1


class MCPConnectionManager:
    """
    Enhanced WebSocket connection manager for MCP with reconnection logic and message queuing.
    """
    
    def __init__(self):
        """Initialize the connection manager."""
        # Active connections
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Subscriptions
        self.provider_subscriptions: Dict[str, Set[str]] = {}
        self.user_connections: Dict[str, Set[str]] = {}
        
        # Message queues for offline clients
        self.message_queues: Dict[str, List[QueuedMessage]] = {}
        
        # Last activity timestamps for heartbeat tracking
        self.last_activity: Dict[str, datetime] = {}
        
        # Connection state
        self.connection_state: Dict[str, str] = {}  # connected, disconnected, reconnecting
        
        # Locks
        self._lock = asyncio.Lock()
        
        # Background tasks
        self._queue_cleanup_task = None
        self._heartbeat_task = None
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self) -> None:
        """Start background tasks for queue cleanup and heartbeats."""
        # Check if tasks should be disabled via environment variable
        import os
        if os.environ.get("DISABLE_MCP_WEBSOCKET_TASKS") == "1":
            logger.info("MCP WebSocket background tasks disabled via environment variable")
            return
            
        if self._queue_cleanup_task is None or self._queue_cleanup_task.done():
            self._queue_cleanup_task = asyncio.create_task(self._cleanup_message_queues())
            logger.info("Started message queue cleanup task")
        
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._send_heartbeats())
            logger.info("Started heartbeat task")
    
    async def connect(self, websocket: WebSocket, client_id: str, user_id: str) -> None:
        """
        Connect a WebSocket client with enhanced handling.
        
        Args:
            websocket: The WebSocket connection
            client_id: Unique client identifier
            user_id: User ID
        """
        await websocket.accept()
        
        async with self._lock:
            # Store connection
            self.active_connections[client_id] = websocket
            
            # Update connection state
            previous_state = self.connection_state.get(client_id)
            self.connection_state[client_id] = "connected"
            
            # Track user connections
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(client_id)
            
            # Update last activity
            self.last_activity[client_id] = datetime.utcnow()
            
            # Check if this is a reconnection
            is_reconnection = previous_state == "disconnected"
        
        logger.info(f"Client {client_id} {'reconnected' if is_reconnection else 'connected'} (user: {user_id})")
        
        # Send welcome message
        await self.send_message(client_id, {
            "type": "connection_established",
            "client_id": client_id,
            "timestamp": datetime.utcnow().isoformat(),
            "reconnected": is_reconnection
        })
        
        # Process queued messages if this is a reconnection
        if is_reconnection:
            await self._process_queued_messages(client_id)
    
    def disconnect(self, client_id: str) -> None:
        """
        Disconnect a WebSocket client but maintain subscriptions for reconnection.
        
        Args:
            client_id: Client identifier to disconnect
        """
        async def _async_disconnect():
            async with self._lock:
                # Remove from active connections but keep subscriptions
                if client_id in self.active_connections:
                    del self.active_connections[client_id]
                
                # Update connection state
                self.connection_state[client_id] = "disconnected"
                
                # Keep track of last activity
                self.last_activity[client_id] = datetime.utcnow()
            
            logger.info(f"Client {client_id} disconnected (subscriptions maintained)")
        
        # Schedule the async operation
        asyncio.create_task(_async_disconnect())
    
    def remove_client(self, client_id: str) -> None:
        """
        Completely remove a client and all its subscriptions.
        
        Args:
            client_id: Client identifier to remove
        """
        async def _async_remove():
            async with self._lock:
                # Remove from active connections
                if client_id in self.active_connections:
                    del self.active_connections[client_id]
                
                # Remove from provider subscriptions
                for provider_id, subscribers in self.provider_subscriptions.items():
                    if client_id in subscribers:
                        subscribers.remove(client_id)
                
                # Remove from user connections
                for user_id, connections in self.user_connections.items():
                    if client_id in connections:
                        connections.remove(client_id)
                        break
                
                # Remove message queue
                if client_id in self.message_queues:
                    del self.message_queues[client_id]
                
                # Remove connection state
                if client_id in self.connection_state:
                    del self.connection_state[client_id]
                
                # Remove last activity
                if client_id in self.last_activity:
                    del self.last_activity[client_id]
            
            logger.info(f"Client {client_id} completely removed")
        
        # Schedule the async operation
        asyncio.create_task(_async_remove())
    
    def subscribe_to_provider(self, client_id: str, provider_id: str) -> None:
        """
        Subscribe a client to a provider.
        
        Args:
            client_id: Client identifier
            provider_id: Provider identifier
        """
        async def _async_subscribe():
            async with self._lock:
                if provider_id not in self.provider_subscriptions:
                    self.provider_subscriptions[provider_id] = set()
                
                self.provider_subscriptions[provider_id].add(client_id)
                
                # Update last activity
                self.last_activity[client_id] = datetime.utcnow()
            
            logger.info(f"Client {client_id} subscribed to provider {provider_id}")
        
        # Schedule the async operation
        asyncio.create_task(_async_subscribe())
    
    def unsubscribe_from_provider(self, client_id: str, provider_id: str) -> None:
        """
        Unsubscribe a client from a provider.
        
        Args:
            client_id: Client identifier
            provider_id: Provider identifier
        """
        async def _async_unsubscribe():
            async with self._lock:
                if provider_id in self.provider_subscriptions and client_id in self.provider_subscriptions[provider_id]:
                    self.provider_subscriptions[provider_id].remove(client_id)
                    
                    # Update last activity
                    self.last_activity[client_id] = datetime.utcnow()
                    
                    logger.info(f"Client {client_id} unsubscribed from provider {provider_id}")
        
        # Schedule the async operation
        asyncio.create_task(_async_unsubscribe())
    
    async def broadcast_to_provider_subscribers(
        self, 
        provider_id: str, 
        message: Dict[str, Any],
        priority: int = 0
    ) -> None:
        """
        Broadcast a message to all subscribers of a provider with queuing for offline clients.
        
        Args:
            provider_id: Provider identifier
            message: Message to broadcast
            priority: Message priority (higher = more important)
        """
        if provider_id not in self.provider_subscriptions:
            return
        
        async with self._lock:
            subscribers = list(self.provider_subscriptions[provider_id])
        
        disconnected_clients = []
        
        for client_id in subscribers:
            # Check if client is connected
            is_connected = client_id in self.active_connections
            
            if is_connected:
                # Try to send message
                success = await self.send_message(client_id, message)
                if not success:
                    # Queue message for later delivery
                    await self._queue_message(client_id, message, priority)
                    disconnected_clients.append(client_id)
            else:
                # Queue message for offline client
                await self._queue_message(client_id, message, priority)
    
    async def send_message(self, client_id: str, message: Dict[str, Any]) -> bool:
        """
        Send a message to a specific client.
        
        Args:
            client_id: Client identifier
            message: Message to send
            
        Returns:
            True if message was sent, False otherwise
        """
        if client_id not in self.active_connections:
            return False
        
        try:
            websocket = self.active_connections[client_id]
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps(message))
                
                # Update last activity
                self.last_activity[client_id] = datetime.utcnow()
                
                return True
        except Exception as e:
            logger.error(f"Error sending message to client {client_id}: {str(e)}")
            self.disconnect(client_id)
        
        return False
    
    async def _queue_message(self, client_id: str, message: Dict[str, Any], priority: int = 0) -> None:
        """
        Queue a message for later delivery.
        
        Args:
            client_id: Client identifier
            message: Message to queue
            priority: Message priority (higher = more important)
        """
        async with self._lock:
            if client_id not in self.message_queues:
                self.message_queues[client_id] = []
            
            # Add message to queue
            queued_message = QueuedMessage(message, priority)
            self.message_queues[client_id].append(queued_message)
            
            # Enforce queue size limit
            if len(self.message_queues[client_id]) > MAX_QUEUE_SIZE:
                # Sort by priority (highest first) and then by timestamp (newest first)
                self.message_queues[client_id].sort(
                    key=lambda m: (m.priority, m.timestamp), 
                    reverse=True
                )
                
                # Trim queue to max size
                self.message_queues[client_id] = self.message_queues[client_id][:MAX_QUEUE_SIZE]
    
    async def _process_queued_messages(self, client_id: str) -> None:
        """
        Process queued messages for a client.
        
        Args:
            client_id: Client identifier
        """
        if client_id not in self.message_queues:
            return
        
        async with self._lock:
            # Get queued messages
            queued_messages = self.message_queues[client_id]
            
            # Sort by priority (highest first) and then by timestamp (oldest first)
            queued_messages.sort(
                key=lambda m: (m.priority, -m.timestamp.timestamp()), 
                reverse=True
            )
            
            # Clear queue
            self.message_queues[client_id] = []
        
        # Send messages
        sent_count = 0
        failed_count = 0
        
        for queued_message in queued_messages:
            # Skip expired messages
            if queued_message.is_expired():
                continue
            
            # Send message
            success = await self.send_message(client_id, queued_message.message)
            
            if success:
                sent_count += 1
            else:
                failed_count += 1
                # Re-queue message with incremented attempt count
                queued_message.increment_attempts()
                if queued_message.attempts < 3:  # Limit retry attempts
                    await self._queue_message(
                        client_id, 
                        queued_message.message, 
                        queued_message.priority
                    )
        
        if sent_count > 0 or failed_count > 0:
            logger.info(f"Processed queued messages for client {client_id}: {sent_count} sent, {failed_count} failed")
    
    async def _cleanup_message_queues(self) -> None:
        """
        Periodically clean up expired messages from queues.
        """
        while True:
            try:
                await asyncio.sleep(QUEUE_CLEANUP_INTERVAL)
                
                async with self._lock:
                    # Process each client's queue
                    for client_id, queue in list(self.message_queues.items()):
                        # Remove expired messages
                        original_size = len(queue)
                        queue = [msg for msg in queue if not msg.is_expired()]
                        self.message_queues[client_id] = queue
                        
                        # Remove empty queues
                        if not queue:
                            del self.message_queues[client_id]
                        elif original_size != len(queue):
                            logger.debug(f"Cleaned up {original_size - len(queue)} expired messages for client {client_id}")
                
                # Check for stale clients
                await self._cleanup_stale_clients()
            
            except asyncio.CancelledError:
                logger.info("Message queue cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in message queue cleanup: {str(e)}")
                await asyncio.sleep(10)  # Wait a bit before retrying
    
    async def _cleanup_stale_clients(self) -> None:
        """
        Clean up stale clients that haven't reconnected in a long time.
        """
        now = datetime.utcnow()
        stale_threshold = now - timedelta(hours=24)  # 24 hours
        
        async with self._lock:
            # Find stale clients
            stale_clients = []
            for client_id, last_active in self.last_activity.items():
                if last_active < stale_threshold and self.connection_state.get(client_id) == "disconnected":
                    stale_clients.append(client_id)
            
            # Remove stale clients
            for client_id in stale_clients:
                # Remove from active connections
                if client_id in self.active_connections:
                    del self.active_connections[client_id]
                
                # Remove from provider subscriptions
                for provider_id, subscribers in self.provider_subscriptions.items():
                    if client_id in subscribers:
                        subscribers.remove(client_id)
                
                # Remove from user connections
                for user_id, connections in self.user_connections.items():
                    if client_id in connections:
                        connections.remove(client_id)
                        break
                
                # Remove message queue
                if client_id in self.message_queues:
                    del self.message_queues[client_id]
                
                # Remove connection state
                if client_id in self.connection_state:
                    del self.connection_state[client_id]
                
                # Remove last activity
                if client_id in self.last_activity:
                    del self.last_activity[client_id]
                
                logger.info(f"Removed stale client {client_id}")
    
    async def _send_heartbeats(self) -> None:
        """
        Periodically send heartbeats to connected clients.
        """
        while True:
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                
                now = datetime.utcnow()
                heartbeat_message = {
                    "type": "heartbeat",
                    "timestamp": now.isoformat()
                }
                
                async with self._lock:
                    # Send heartbeats to all connected clients
                    for client_id, websocket in list(self.active_connections.items()):
                        try:
                            if websocket.client_state == WebSocketState.CONNECTED:
                                await websocket.send_text(json.dumps(heartbeat_message))
                                logger.debug(f"Sent heartbeat to client {client_id}")
                        except Exception as e:
                            logger.warning(f"Error sending heartbeat to client {client_id}: {str(e)}")
                            self.disconnect(client_id)
                    
                    # Check for clients that haven't responded
                    for client_id, last_active in list(self.last_activity.items()):
                        if client_id in self.active_connections:
                            inactive_time = (now - last_active).total_seconds()
                            if inactive_time > HEARTBEAT_TIMEOUT:
                                logger.warning(f"Client {client_id} timed out after {inactive_time:.1f}s of inactivity")
                                self.disconnect(client_id)
            
            except asyncio.CancelledError:
                logger.info("Heartbeat task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in heartbeat task: {str(e)}")
                await asyncio.sleep(10)  # Wait a bit before retrying
    
    def get_provider_subscribers(self, provider_id: str) -> List[str]:
        """
        Get all subscribers for a provider.
        
        Args:
            provider_id: Provider identifier
            
        Returns:
            List of client identifiers
        """
        if provider_id not in self.provider_subscriptions:
            return []
        
        return list(self.provider_subscriptions[provider_id])
    
    def get_user_connections(self, user_id: str) -> List[str]:
        """
        Get all connections for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of client identifiers
        """
        if user_id not in self.user_connections:
            return []
        
        return list(self.user_connections[user_id])
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get connection statistics.
        
        Returns:
            Dictionary with connection statistics
        """
        return {
            "active_connections": len(self.active_connections),
            "provider_subscriptions": {
                provider_id: len(subscribers)
                for provider_id, subscribers in self.provider_subscriptions.items()
            },
            "user_connections": {
                user_id: len(connections)
                for user_id, connections in self.user_connections.items()
            },
            "message_queues": {
                client_id: len(queue)
                for client_id, queue in self.message_queues.items()
            },
            "total_queued_messages": sum(len(queue) for queue in self.message_queues.values())
        }
    
    async def shutdown(self) -> None:
        """
        Shutdown the connection manager and clean up resources.
        """
        # Cancel background tasks
        if self._queue_cleanup_task and not self._queue_cleanup_task.done():
            self._queue_cleanup_task.cancel()
            try:
                await self._queue_cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        close_message = {
            "type": "server_shutdown",
            "message": "Server is shutting down",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        for client_id, websocket in list(self.active_connections.items()):
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(json.dumps(close_message))
                    await websocket.close()
            except Exception:
                pass
        
        # Clear all data
        self.active_connections.clear()
        self.provider_subscriptions.clear()
        self.user_connections.clear()
        self.message_queues.clear()
        self.last_activity.clear()
        self.connection_state.clear()
        
        logger.info("Connection manager shut down")


# Create singleton instance
mcp_manager = MCPConnectionManager()
