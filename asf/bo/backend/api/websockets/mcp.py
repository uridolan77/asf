"""
WebSocket endpoint for MCP provider status updates.

This module provides a WebSocket endpoint for real-time updates
of MCP provider status, metrics, and events.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Set, Any, Optional

from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from starlette.websockets import WebSocketState

from api.auth.dependencies import get_current_user_ws
from models.user import User
from api.services.llm.gateway_service import get_llm_gateway_service
from api.websockets.mcp_manager import mcp_manager
from api.websockets.auth import authenticate_ws_connection, check_mcp_access

logger = logging.getLogger(__name__)

# Use the enhanced MCP connection manager from mcp_manager.py


async def handle_mcp_websocket(
    websocket: WebSocket,
    client_id: str
) -> None:
    """
    Handle WebSocket connections for MCP provider updates with enhanced authentication.

    Args:
        websocket: The WebSocket connection
        client_id: Unique client identifier
    """
    # Authenticate the WebSocket connection
    user = await authenticate_ws_connection(websocket)
    if not user:
        return  # Connection already closed by authenticate_ws_connection

    # Check if user has access to MCP functionality
    has_access = await check_mcp_access(user)
    if not has_access:
        logger.warning(f"User {user.id} does not have access to MCP functionality")
        await websocket.close(code=1008)  # Policy violation
        return

    # Connect to the MCP manager
    await mcp_manager.connect(websocket, client_id, str(user.id))

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                message_type = message.get('type')

                # Handle heartbeat response
                if message_type == 'heartbeat_ack':
                    # Update last activity timestamp
                    continue

                provider_id = message.get('provider_id')

                if not message_type:
                    continue

                # Handle message based on type
                if message_type == 'subscribe' and provider_id:
                    mcp_manager.subscribe_to_provider(client_id, provider_id)

                    # Send acknowledgment
                    await mcp_manager.send_message(client_id, {
                        'type': 'subscription_ack',
                        'provider_id': provider_id,
                        'timestamp': datetime.utcnow().isoformat()
                    })

                    # Send initial status update
                    try:
                        gateway_service = get_llm_gateway_service()
                        status = await gateway_service.get_provider_status(provider_id)

                        await mcp_manager.send_message(client_id, {
                            'type': 'provider_status',
                            'provider_id': provider_id,
                            'data': status,
                            'timestamp': datetime.utcnow().isoformat()
                        })
                    except Exception as e:
                        logger.error(f"Error getting initial status for provider {provider_id}: {str(e)}")

                elif message_type == 'unsubscribe' and provider_id:
                    mcp_manager.unsubscribe_from_provider(client_id, provider_id)

                    # Send acknowledgment
                    await mcp_manager.send_message(client_id, {
                        'type': 'unsubscription_ack',
                        'provider_id': provider_id,
                        'timestamp': datetime.utcnow().isoformat()
                    })

                elif message_type == 'request_status' and provider_id:
                    # Get provider status
                    gateway_service = get_llm_gateway_service()
                    status = await gateway_service.get_provider_status(provider_id)

                    # Send status update
                    await mcp_manager.send_message(client_id, {
                        'type': 'provider_status',
                        'provider_id': provider_id,
                        'data': status,
                        'timestamp': datetime.utcnow().isoformat()
                    })

                elif message_type == 'request_metrics' and provider_id:
                    # Get provider metrics
                    gateway_service = get_llm_gateway_service()
                    metrics = await gateway_service.get_provider_metrics(provider_id)

                    # Send metrics update
                    await mcp_manager.send_message(client_id, {
                        'type': 'provider_metrics',
                        'provider_id': provider_id,
                        'data': metrics,
                        'timestamp': datetime.utcnow().isoformat()
                    })

                elif message_type == 'request_connection_stats':
                    # Get connection statistics
                    stats = mcp_manager.get_connection_stats()

                    # Send connection statistics
                    await mcp_manager.send_message(client_id, {
                        'type': 'connection_stats',
                        'data': stats,
                        'timestamp': datetime.utcnow().isoformat()
                    })

            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received from client {client_id}")
            except Exception as e:
                logger.error(f"Error processing message from client {client_id}: {str(e)}")

    except WebSocketDisconnect:
        mcp_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket handler: {str(e)}")
        mcp_manager.disconnect(client_id)


async def broadcast_provider_status(provider_id: str, status: Dict[str, Any]) -> None:
    """
    Broadcast provider status to all subscribers with enhanced message queuing.

    Args:
        provider_id: Provider identifier
        status: Provider status
    """
    await mcp_manager.broadcast_to_provider_subscribers(
        provider_id,
        {
            'type': 'provider_status',
            'provider_id': provider_id,
            'data': status,
            'timestamp': datetime.utcnow().isoformat()
        },
        priority=2  # Higher priority for status updates
    )


async def broadcast_provider_metrics(provider_id: str, metrics: Dict[str, Any]) -> None:
    """
    Broadcast provider metrics to all subscribers with enhanced message queuing.

    Args:
        provider_id: Provider identifier
        metrics: Provider metrics
    """
    await mcp_manager.broadcast_to_provider_subscribers(
        provider_id,
        {
            'type': 'provider_metrics',
            'provider_id': provider_id,
            'data': metrics,
            'timestamp': datetime.utcnow().isoformat()
        },
        priority=1  # Medium priority for metrics updates
    )


async def broadcast_provider_event(provider_id: str, event_type: str, event_data: Dict[str, Any]) -> None:
    """
    Broadcast provider event to all subscribers with enhanced message queuing.

    Args:
        provider_id: Provider identifier
        event_type: Event type
        event_data: Event data
    """
    # Determine priority based on event type
    priority = 0  # Default priority
    if event_type in ['error', 'circuit_breaker_open', 'circuit_breaker_closed']:
        priority = 3  # Highest priority for critical events
    elif event_type in ['request', 'response']:
        priority = 1  # Medium priority for request/response events

    await mcp_manager.broadcast_to_provider_subscribers(
        provider_id,
        {
            'type': 'provider_event',
            'provider_id': provider_id,
            'event_type': event_type,
            'data': event_data,
            'timestamp': datetime.utcnow().isoformat()
        },
        priority=priority
    )


async def shutdown_mcp_websockets():
    """
    Shutdown MCP WebSocket connections gracefully.
    """
    await mcp_manager.shutdown()
