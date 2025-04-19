"""
WebSocket endpoint for MCP provider status updates.

This module provides a WebSocket endpoint for real-time updates
of MCP provider status, metrics, and events.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Set, Any, Optional

from fastapi import WebSocket, WebSocketDisconnect, Depends
from starlette.websockets import WebSocketState

from api.auth.dependencies import get_current_user_ws
from models.user import User
from api.services.llm.gateway_service import get_llm_gateway_service as get_gateway_service

logger = logging.getLogger(__name__)

# Store active connections
class ConnectionManager:
    """
    Manages WebSocket connections and subscriptions.
    """

    def __init__(self):
        """Initialize the connection manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.provider_subscriptions: Dict[str, Set[str]] = {}
        self.user_connections: Dict[str, Set[str]] = {}

    async def connect(self, websocket: WebSocket, client_id: str, user_id: str) -> None:
        """
        Connect a WebSocket client.

        Args:
            websocket: The WebSocket connection
            client_id: Unique client identifier
            user_id: User ID
        """
        await websocket.accept()
        self.active_connections[client_id] = websocket

        # Track user connections
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(client_id)

        logger.info(f"Client {client_id} connected (user: {user_id})")

    def disconnect(self, client_id: str) -> None:
        """
        Disconnect a WebSocket client.

        Args:
            client_id: Client identifier to disconnect
        """
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

        logger.info(f"Client {client_id} disconnected")

    def subscribe_to_provider(self, client_id: str, provider_id: str) -> None:
        """
        Subscribe a client to a provider.

        Args:
            client_id: Client identifier
            provider_id: Provider identifier
        """
        if provider_id not in self.provider_subscriptions:
            self.provider_subscriptions[provider_id] = set()

        self.provider_subscriptions[provider_id].add(client_id)
        logger.info(f"Client {client_id} subscribed to provider {provider_id}")

    def unsubscribe_from_provider(self, client_id: str, provider_id: str) -> None:
        """
        Unsubscribe a client from a provider.

        Args:
            client_id: Client identifier
            provider_id: Provider identifier
        """
        if provider_id in self.provider_subscriptions and client_id in self.provider_subscriptions[provider_id]:
            self.provider_subscriptions[provider_id].remove(client_id)
            logger.info(f"Client {client_id} unsubscribed from provider {provider_id}")

    async def broadcast_to_provider_subscribers(self, provider_id: str, message: Dict[str, Any]) -> None:
        """
        Broadcast a message to all subscribers of a provider.

        Args:
            provider_id: Provider identifier
            message: Message to broadcast
        """
        if provider_id not in self.provider_subscriptions:
            return

        disconnected_clients = []

        for client_id in self.provider_subscriptions[provider_id]:
            if client_id in self.active_connections:
                try:
                    websocket = self.active_connections[client_id]
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error sending message to client {client_id}: {str(e)}")
                    disconnected_clients.append(client_id)
            else:
                disconnected_clients.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)

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
                return True
        except Exception as e:
            logger.error(f"Error sending message to client {client_id}: {str(e)}")
            self.disconnect(client_id)

        return False

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


# Create singleton instance
manager = ConnectionManager()


async def handle_mcp_websocket(
    websocket: WebSocket,
    client_id: str,
    user: User = Depends(get_current_user_ws)
) -> None:
    """
    Handle WebSocket connections for MCP provider updates.

    Args:
        websocket: The WebSocket connection
        client_id: Unique client identifier
        user: Authenticated user
    """
    await manager.connect(websocket, client_id, str(user.id))

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                message_type = message.get('type')
                provider_id = message.get('provider_id')

                if not message_type:
                    continue

                # Handle message based on type
                if message_type == 'subscribe' and provider_id:
                    manager.subscribe_to_provider(client_id, provider_id)

                    # Send acknowledgment
                    await manager.send_message(client_id, {
                        'type': 'subscription_ack',
                        'provider_id': provider_id,
                        'timestamp': datetime.utcnow().isoformat()
                    })

                elif message_type == 'unsubscribe' and provider_id:
                    manager.unsubscribe_from_provider(client_id, provider_id)

                    # Send acknowledgment
                    await manager.send_message(client_id, {
                        'type': 'unsubscription_ack',
                        'provider_id': provider_id,
                        'timestamp': datetime.utcnow().isoformat()
                    })

                elif message_type == 'request_status' and provider_id:
                    # Get provider status
                    gateway_service = get_gateway_service()
                    status = await gateway_service.get_provider_status(provider_id)

                    # Send status update
                    await manager.send_message(client_id, {
                        'type': 'provider_status',
                        'provider_id': provider_id,
                        'data': status,
                        'timestamp': datetime.utcnow().isoformat()
                    })

                elif message_type == 'request_metrics' and provider_id:
                    # Get provider metrics
                    gateway_service = get_gateway_service()
                    metrics = await gateway_service.get_provider_metrics(provider_id)

                    # Send metrics update
                    await manager.send_message(client_id, {
                        'type': 'provider_metrics',
                        'provider_id': provider_id,
                        'data': metrics,
                        'timestamp': datetime.utcnow().isoformat()
                    })

            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received from client {client_id}")
            except Exception as e:
                logger.error(f"Error processing message from client {client_id}: {str(e)}")

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket handler: {str(e)}")
        manager.disconnect(client_id)


async def broadcast_provider_status(provider_id: str, status: Dict[str, Any]) -> None:
    """
    Broadcast provider status to all subscribers.

    Args:
        provider_id: Provider identifier
        status: Provider status
    """
    await manager.broadcast_to_provider_subscribers(provider_id, {
        'type': 'provider_status',
        'provider_id': provider_id,
        'data': status,
        'timestamp': datetime.utcnow().isoformat()
    })


async def broadcast_provider_metrics(provider_id: str, metrics: Dict[str, Any]) -> None:
    """
    Broadcast provider metrics to all subscribers.

    Args:
        provider_id: Provider identifier
        metrics: Provider metrics
    """
    await manager.broadcast_to_provider_subscribers(provider_id, {
        'type': 'provider_metrics',
        'provider_id': provider_id,
        'data': metrics,
        'timestamp': datetime.utcnow().isoformat()
    })


async def broadcast_provider_event(provider_id: str, event_type: str, event_data: Dict[str, Any]) -> None:
    """
    Broadcast provider event to all subscribers.

    Args:
        provider_id: Provider identifier
        event_type: Event type
        event_data: Event data
    """
    await manager.broadcast_to_provider_subscribers(provider_id, {
        'type': 'provider_event',
        'provider_id': provider_id,
        'event_type': event_type,
        'data': event_data,
        'timestamp': datetime.utcnow().isoformat()
    })
